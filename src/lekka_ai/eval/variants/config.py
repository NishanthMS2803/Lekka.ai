"""Configuration models and ablation switches for evaluation engine variants."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VariantType = Literal["static", "dynamic", "dynamic_replan"]
QuantToolsMode = Literal["quant_tools", "run_python_only"]


class VariantConfig(BaseModel):
    """Configuration options and ablation switches for engine variants.

    Attributes:
        variant_type: Orchestration architecture style:
            - 'static' (V3): all 5 specialists execute in parallel; 1 round.
            - 'dynamic' (V4): planner selects agents based on query; 1 round.
            - 'dynamic_replan' (V5): planner + gap check replanning + verifier.
        verifier: Ablation switch enabling or disabling claim support verification.
        risk_agent: Ablation switch enabling or disabling the risk specialist.
        memory: Ablation switch enabling or disabling facts cache / memory tools.
        quant_tools_mode: 'quant_tools' (standard quant tool suite) vs
            'run_python_only' (restricting financial math strictly to Python sandbox).
        max_rounds: Maximum number of orchestration wave rounds to execute.
        enable_gap_check: Whether to execute gap check audit between waves.
        risk_wave: Whether to ensure a dedicated risk assessment wave runs.
    """

    model_config = ConfigDict(extra="ignore")

    variant_type: VariantType = Field(
        default="dynamic",
        description="Architecture style ('static', 'dynamic', 'dynamic_replan')",
    )
    verifier: bool = Field(
        default=True,
        description="Ablation switch: source support verification on/off",
    )
    risk_agent: bool = Field(
        default=True,
        description="Ablation switch: risk specialist on/off",
    )
    memory: bool = Field(
        default=True,
        description="Ablation switch: memory / recall_facts tool on/off",
    )
    quant_tools_mode: QuantToolsMode = Field(
        default="quant_tools",
        description="Ablation switch: 'quant_tools' vs 'run_python_only'",
    )
    max_rounds: int = Field(
        default=1,
        description="Maximum orchestration rounds permitted",
    )
    enable_gap_check: bool = Field(
        default=False,
        description="Whether to run gap-check audit and re-planning",
    )
    risk_wave: bool = Field(
        default=False,
        description="Whether to trigger a dedicated risk audit wave",
    )
