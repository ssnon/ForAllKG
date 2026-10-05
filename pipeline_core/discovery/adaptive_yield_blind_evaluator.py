from __future__ import annotations

import json
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.adaptive_yield_evaluation import AdaptiveYieldBlindPacket


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


BlindPreference = Literal["ARM_A", "ARM_B", "TIE", "UNCLEAR"]


class BlindYieldDimensionDraft(StrictModel):
    dimension_id: str
    preference: BlindPreference
    rationale: str = Field(min_length=1)


class BlindYieldComparisonDraft(StrictModel):
    comparison_alias: str
    dimensions: list[BlindYieldDimensionDraft]


class BlindYieldEvaluationDraft(StrictModel):
    comparisons: list[BlindYieldComparisonDraft]


SYSTEM_PROMPT = """Blindly compare scientific hypothesis portfolios.
The arm identities are hidden. Do not infer which system produced an arm.
Candidate count is NOT a quality signal. Do not compute an overall score,
winner, ranking, or tier. Judge each requested dimension independently.
An empty arm can be preferred only when abstention is scientifically more
appropriate than the supplied alternative; do not reward mere quantity.
Use ARM_A, ARM_B, TIE, or UNCLEAR."""


def build_prompt(packet: AdaptiveYieldBlindPacket) -> str:
    return (
        "ADAPTIVE YIELD BLIND EVALUATION\n"
        "================================\n"
        + json.dumps(packet.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n\nReturn every comparison and every requested dimension exactly once."
    )


def validate_draft(packet: AdaptiveYieldBlindPacket, draft: BlindYieldEvaluationDraft) -> None:
    expected_comparisons = {x.comparison_alias for x in packet.comparisons}
    observed_comparisons = {x.comparison_alias for x in draft.comparisons}
    if expected_comparisons != observed_comparisons:
        raise ValueError("blind evaluation comparison set mismatch")
    expected_dimensions = set(packet.dimensions)
    for row in draft.comparisons:
        observed = [x.dimension_id for x in row.dimensions]
        if len(observed) != len(set(observed)):
            raise ValueError("duplicate blind evaluation dimension")
        if set(observed) != expected_dimensions:
            raise ValueError("blind evaluation dimension set mismatch")


def report_payload(packet: AdaptiveYieldBlindPacket, draft: BlindYieldEvaluationDraft, *, model: str) -> dict:
    validate_draft(packet, draft)
    return {
        "schema_version": "adaptive-yield-blind-evaluation-v1",
        "packet_id": packet.packet_id,
        "model": model,
        "comparisons": [
            {
                **row.model_dump(mode="json"),
                "preference_counts": dict(Counter(x.preference for x in row.dimensions)),
            }
            for row in draft.comparisons
        ],
        "candidate_count_treated_as_quality_signal": False,
        "overall_score_computed": False,
        "overall_winner_selected": False,
        "scientific_quality_ranking_performed": False,
        "production_selection_changed": False,
    }
