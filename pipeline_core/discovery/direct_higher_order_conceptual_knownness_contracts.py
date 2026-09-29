from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ConceptualKnownnessLevelName = Literal[
    "L1_BROAD",
    "L2_INTERMEDIATE",
    "L3_EXACT",
]


class ConceptualKnownnessLevelResult(StrictModel):
    level: ConceptualKnownnessLevelName
    abstraction_text: str
    knownness_class: str
    retrieval_status: str
    search_bounded: bool = True
    sufficient_coverage: bool | None = None
    source_profile_path: str | None = None


class DirectHigherOrderConceptualKnownnessRecord(StrictModel):
    schema_version: Literal[
        "direct-higher-order-conceptual-knownness-record-v1"
    ] = "direct-higher-order-conceptual-knownness-record-v1"

    hypothesis_id: str
    direct_context_id: str | None = None
    direct_topology_id: str | None = None
    first_gap_level: ConceptualKnownnessLevelName | None = None
    levels: list[ConceptualKnownnessLevelResult] = Field(default_factory=list)

    diagnostic_only: Literal[True] = True
    novelty_authority_created: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_levels(self):
        names = [row.level for row in self.levels]
        if len(names) != len(set(names)):
            raise ValueError("duplicate conceptual knownness level")
        return self


class DirectHigherOrderConceptualKnownnessProfile(StrictModel):
    schema_version: Literal[
        "direct-higher-order-conceptual-knownness-profile-v1"
    ] = "direct-higher-order-conceptual-knownness-profile-v1"

    profile_id: str
    profile_sha256: str
    source_bundle_id: str
    records: list[DirectHigherOrderConceptualKnownnessRecord] = Field(
        default_factory=list
    )

    diagnostic_only: Literal[True] = True
    candidate_survival_authority: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    production_selection_authority: Literal[False] = False


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return prefix + ":" + hashlib.sha256(raw).hexdigest()[:20]


def _sha256(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def build_conceptual_knownness_profile(
    *,
    source_bundle_id: str,
    records: list[DirectHigherOrderConceptualKnownnessRecord],
) -> DirectHigherOrderConceptualKnownnessProfile:
    ordered = sorted(
        records,
        key=lambda row: (
            row.direct_context_id or "",
            row.hypothesis_id,
        ),
    )

    profile_id = _stable_id(
        "direct_higher_order_conceptual_knownness_profile",
        source_bundle_id,
        *[
            (
                row.hypothesis_id
                + ":"
                + str(row.first_gap_level or "")
            )
            for row in ordered
        ],
    )

    base = {
        "schema_version":
            "direct-higher-order-conceptual-knownness-profile-v1",
        "profile_id":
            profile_id,
        "source_bundle_id":
            source_bundle_id,
        "records": [
            row.model_dump(mode="json")
            for row in ordered
        ],
        "diagnostic_only":
            True,
        "candidate_survival_authority":
            False,
        "novelty_authority_created":
            False,
        "production_selection_authority":
            False,
    }

    return DirectHigherOrderConceptualKnownnessProfile(
        **base,
        profile_sha256=_sha256(base),
    )


__all__ = [
    "ConceptualKnownnessLevelResult",
    "DirectHigherOrderConceptualKnownnessRecord",
    "DirectHigherOrderConceptualKnownnessProfile",
    "build_conceptual_knownness_profile",
]
