"""Verification package auditing claim grounding, consistency, and validity."""

from lekka_ai.verify.checks import (
    CheckResult,
    ChecksReport,
    CheckStatus,
    check_background_not_sole_support,
    check_calculation_recomputation,
    check_evidence_dates_cutoff,
    check_interpretation_supporting_ids,
    verify_deterministic_checks,
)
from lekka_ai.verify.conflicts import (
    Conflict,
    detect_conflicts,
    normalize_metric,
    normalize_period,
)
from lekka_ai.verify.facts_cache import (
    FACT_TTL_POLICIES,
    FactCacheRepository,
    RecalledFact,
    RecallFactsOutput,
    compute_expires_at,
    get_fact_cache_repository,
    get_recalled_facts,
    get_ttl_for_fact_type,
    store_verified_claims_in_cache,
)
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
    "CheckResult",
    "CheckStatus",
    "ChecksReport",
    "ClaimSupportCheck",
    "Conflict",
    "FACT_TTL_POLICIES",
    "FactCacheRepository",
    "RecallFactsOutput",
    "RecalledFact",
    "SupportVerdict",
    "check_background_not_sole_support",
    "check_calculation_recomputation",
    "check_evidence_dates_cutoff",
    "check_interpretation_supporting_ids",
    "compute_expires_at",
    "detect_conflicts",
    "fetch_claim_evidence_passage",
    "get_fact_cache_repository",
    "get_recalled_facts",
    "get_ttl_for_fact_type",
    "normalize_metric",
    "normalize_period",
    "store_verified_claims_in_cache",
    "verify_claim_support",
    "verify_deterministic_checks",
    "verify_support",
]
