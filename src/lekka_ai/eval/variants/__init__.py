"""Engine-backed evaluation variants and ablation switches (V3, V4, V5)."""

from __future__ import annotations

from lekka_ai.eval.variants.base import (
    BaseEngineVariant,
    VariantConfig,
    apply_ablation_to_profile,
    collect_claims_from_state,
)
from lekka_ai.eval.variants.v3_static import V3StaticVariant
from lekka_ai.eval.variants.v4_dynamic import V4DynamicVariant
from lekka_ai.eval.variants.v5_dynamic import V5DynamicVariant

__all__ = [
    "BaseEngineVariant",
    "VariantConfig",
    "apply_ablation_to_profile",
    "collect_claims_from_state",
    "V3StaticVariant",
    "V4DynamicVariant",
    "V5DynamicVariant",
]
