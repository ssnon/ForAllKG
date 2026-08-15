from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dac_her.hypothesis_trend_compiler import (
    USE_TO_LANE,
    required_companion_uses,
)
from dac_her.hypothesis_trend_contracts import TrendReferenceUse
from dac_her.hypothesis_trend_input import (
    HYPOTHESIS_TREND_INPUT_CONTRACT_SEMANTICS_ID,
    HypothesisTrendInputView,
    TrendAwareHypothesisInput,
    verify_trend_aware_input_sources,
)


HYPOTHESIS_TREND_MAKER_EXPOSURE_SEMANTICS_ID = (
    "hypothesis_trend_maker_exposure_v1_alpha4c5d"
)

LANE_TO_USE: dict[str, TrendReferenceUse] = {
    lane: use_role for use_role, lane in USE_TO_LANE.items()
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _sorted_unique(values: list[str]) -> list[str]:
    return sorted({
        str(value)
        for value in values
        if str(value).strip()
    })


class TrendMakerExposurePolicy(StrictModel):
    source_input_contract_immutable: Literal[True] = True
    source_view_maker_selectable_preserved_false: Literal[True] = True
    source_view_causal_use_preserved_false: Literal[True] = True
    source_view_universal_use_preserved_false: Literal[True] = True
    activation_does_not_mutate_source_views: Literal[True] = True
    only_exact_projected_view_ids_exposed: Literal[True] = True
    use_role_derived_from_lane: Literal[True] = True
    limitation_companions_precomputed: Literal[True] = True
    trend_numeric_values_exposed: Literal[False] = False
    causality_authorized_by_activation: Literal[False] = False
    universal_relation_authorized_by_activation: Literal[False] = False
    unknown_context_fill_allowed: Literal[False] = False
    majority_direction_vote_allowed: Literal[False] = False
    external_novelty_claims_authorized: Literal[False] = False
    experimental_protocols_authorized: Literal[False] = False


class TrendMakerRequiredCompanion(StrictModel):
    use_role: TrendReferenceUse
    view_id: str


class TrendMakerExposedView(StrictModel):
    schema_version: Literal[
        "trend-maker-exposed-view-v1"
    ] = "trend-maker-exposed-view-v1"

    view_id: str
    grounding_id: str
    relation_id: str
    lane: str
    allowed_use_role: TrendReferenceUse

    cross_context_status: str
    support_role: str
    independent_variable_key: str
    dependent_observable_key: str
    control_family: str
    observable_semantics: str

    paper_ids: list[str] = Field(default_factory=list)
    directions: list[str] = Field(default_factory=list)
    shapes: list[str] = Field(default_factory=list)
    evidence_kinds: list[str] = Field(default_factory=list)
    evidence_bases: list[str] = Field(default_factory=list)
    differentiating_dimensions: list[str] = Field(default_factory=list)
    unresolved_dimensions: list[str] = Field(default_factory=list)

    association_only: bool
    requires_context_qualification: bool
    requires_verification: bool
    directional_cross_paper_premise_allowed: bool

    required_companions: list[TrendMakerRequiredCompanion] = Field(
        default_factory=list
    )

    selectable_by_maker: Literal[True] = True
    causal_use_allowed: Literal[False] = False
    universal_use_allowed: Literal[False] = False
    numeric_values_exposed: Literal[False] = False

    @model_validator(mode="after")
    def _consistency(self) -> "TrendMakerExposedView":
        expected_role = LANE_TO_USE.get(self.lane)
        if expected_role is None:
            raise ValueError(
                f"Unknown Trend Maker exposure lane: {self.lane!r}."
            )
        if self.allowed_use_role != expected_role:
            raise ValueError(
                "Trend Maker exposure use role does not match source lane."
            )

        for name in (
            "paper_ids",
            "directions",
            "shapes",
            "evidence_kinds",
            "evidence_bases",
            "differentiating_dimensions",
            "unresolved_dimensions",
        ):
            values = getattr(self, name)
            if values != sorted(set(values)):
                raise ValueError(f"{name} must be sorted and unique.")

        companion_rows = [
            (row.use_role, row.view_id)
            for row in self.required_companions
        ]
        if companion_rows != sorted(set(companion_rows)):
            raise ValueError(
                "required_companions must be sorted and unique."
            )
        return self


class TrendMakerExposure(StrictModel):
    schema_version: Literal[
        "trend-maker-exposure-v1"
    ] = "trend-maker-exposure-v1"

    exposure_id: str
    exposure_sha256: str
    semantics_id: str

    source_input_id: str
    source_input_sha256: str
    source_input_contract_semantics_id: str
    source_input_semantics_id: str
    domain_profile_id: str
    corpus_id: str

    views: list[TrendMakerExposedView] = Field(default_factory=list)
    lane_counts: dict[str, int] = Field(default_factory=dict)
    policy: TrendMakerExposurePolicy = Field(
        default_factory=TrendMakerExposurePolicy
    )

    @model_validator(mode="after")
    def _consistency(self) -> "TrendMakerExposure":
        if self.semantics_id != (
            HYPOTHESIS_TREND_MAKER_EXPOSURE_SEMANTICS_ID
        ):
            raise ValueError("Trend Maker exposure semantics mismatch.")
        if self.source_input_contract_semantics_id != (
            HYPOTHESIS_TREND_INPUT_CONTRACT_SEMANTICS_ID
        ):
            raise ValueError(
                "Trend Maker exposure requires the frozen alpha4c.5b input contract."
            )

        ids = [row.view_id for row in self.views]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate exposed Trend view_id.")

        expected_counts = dict(sorted(Counter(
            row.lane for row in self.views
        ).items()))
        if self.lane_counts != expected_counts:
            raise ValueError("Trend Maker exposure lane_counts mismatch.")

        expected_id = _stable_id(
            "trend_maker_exposure",
            self.semantics_id,
            self.source_input_sha256,
        )
        if self.exposure_id != expected_id:
            raise ValueError("Trend Maker exposure_id is not stable.")

        payload = self.model_dump(mode="json")
        observed_sha = str(payload.pop("exposure_sha256", ""))
        expected_sha = _sha256_json(payload)
        if observed_sha != expected_sha:
            raise ValueError("Trend Maker exposure SHA mismatch.")
        return self


def _assert_frozen_source_policy(source: TrendAwareHypothesisInput) -> None:
    policy = source.policy
    frozen_false_fields = (
        "maker_consumption_enabled",
        "prompt_modified",
        "compiler_modified",
        "validator_modified",
        "runtime_modified",
        "llm_calls_allowed",
        "causality_authorized_by_trend_input",
        "universal_relation_authorized_by_trend_input",
        "unknown_context_fill_allowed",
        "majority_direction_vote_allowed",
    )
    for name in frozen_false_fields:
        if getattr(policy, name) is not False:
            raise ValueError(
                "alpha4c.5d activation refuses drift in frozen alpha4c.5b "
                f"policy field {name!r}."
            )

    for view in source.trend_views:
        if view.maker_selectable is not False:
            raise ValueError(
                "alpha4c.5d must not mutate 5b maker_selectable."
            )
        if view.causal_use_allowed is not False:
            raise ValueError(
                "alpha4c.5d must not mutate 5b causal_use_allowed."
            )
        if view.universal_use_allowed is not False:
            raise ValueError(
                "alpha4c.5d must not mutate 5b universal_use_allowed."
            )


def build_trend_maker_exposure(
    source: TrendAwareHypothesisInput,
) -> TrendMakerExposure:
    verify_trend_aware_input_sources(source)
    _assert_frozen_source_policy(source)

    by_grounding_lane: dict[tuple[str, str], HypothesisTrendInputView] = {
        (view.grounding_id, view.lane): view
        for view in source.trend_views
    }

    exposed: list[TrendMakerExposedView] = []
    for view in source.trend_views:
        use_role = LANE_TO_USE.get(view.lane)
        if use_role is None:
            raise ValueError(
                f"No alpha4c.5c use role exists for lane {view.lane!r}."
            )

        required_roles: list[TrendReferenceUse] = []
        required_companions: list[TrendMakerRequiredCompanion] = []
        if use_role in {
            "positive_empirical_support",
            "cross_paper_empirical_support",
        }:
            required_roles = sorted(required_companion_uses(view))
            for role in required_roles:
                companion_lane = USE_TO_LANE[role]
                companion = by_grounding_lane.get(
                    (view.grounding_id, companion_lane)
                )
                if companion is None:
                    raise ValueError(
                        "Frozen 5b input is missing a limitation companion "
                        f"required by 5c: grounding={view.grounding_id!r}, "
                        f"role={role!r}, lane={companion_lane!r}."
                    )
                required_companions.append(
                    TrendMakerRequiredCompanion(
                        use_role=role,
                        view_id=companion.view_id,
                    )
                )

        exposed.append(
            TrendMakerExposedView(
                view_id=view.view_id,
                grounding_id=view.grounding_id,
                relation_id=view.relation_id,
                lane=view.lane,
                allowed_use_role=use_role,
                cross_context_status=view.cross_context_status,
                support_role=view.support_role,
                independent_variable_key=view.independent_variable_key,
                dependent_observable_key=view.dependent_observable_key,
                control_family=view.control_family,
                observable_semantics=view.observable_semantics,
                paper_ids=list(view.paper_ids),
                directions=list(view.directions),
                shapes=list(view.shapes),
                evidence_kinds=list(view.evidence_kinds),
                evidence_bases=list(view.evidence_bases),
                differentiating_dimensions=list(
                    view.differentiating_dimensions
                ),
                unresolved_dimensions=list(view.unresolved_dimensions),
                association_only=bool(view.association_only_result_ids),
                requires_context_qualification=
                    view.requires_context_qualification,
                requires_verification=view.requires_verification,
                directional_cross_paper_premise_allowed=
                    view.directional_cross_paper_premise_allowed,
                required_companions=sorted(
                    required_companions,
                    key=lambda row: (row.use_role, row.view_id),
                ),
                selectable_by_maker=True,
                causal_use_allowed=False,
                universal_use_allowed=False,
                numeric_values_exposed=False,
            )
        )

    exposure_id = _stable_id(
        "trend_maker_exposure",
        HYPOTHESIS_TREND_MAKER_EXPOSURE_SEMANTICS_ID,
        source.input_sha256,
    )
    payload = {
        "schema_version": "trend-maker-exposure-v1",
        "exposure_id": exposure_id,
        "semantics_id": HYPOTHESIS_TREND_MAKER_EXPOSURE_SEMANTICS_ID,
        "source_input_id": source.input_id,
        "source_input_sha256": source.input_sha256,
        "source_input_contract_semantics_id": source.contract_semantics_id,
        "source_input_semantics_id": source.input_semantics_id,
        "domain_profile_id": source.domain_profile_id,
        "corpus_id": source.corpus_id,
        "views": [row.model_dump(mode="json") for row in exposed],
        "lane_counts": dict(sorted(Counter(
            row.lane for row in exposed
        ).items())),
        "policy": TrendMakerExposurePolicy().model_dump(mode="json"),
    }
    payload["exposure_sha256"] = _sha256_json(payload)
    return TrendMakerExposure(**payload)


def verify_trend_maker_exposure(
    source: TrendAwareHypothesisInput,
    exposure: TrendMakerExposure,
) -> None:
    expected = build_trend_maker_exposure(source)
    if exposure.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise ValueError(
            "Trend Maker exposure is not the exact deterministic activation "
            "of the supplied alpha4c.5b input."
        )
