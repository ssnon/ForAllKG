from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dac_her.cross_context_trend import (
    CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID,
    CrossContextTrendAssessment,
    PairwiseTrendContrast,
    TrendContextDimension,
    TrendContextProfile,
)
from dac_her.cross_context_trend_assessment import (
    CROSS_CONTEXT_TREND_ASSESSMENT_SEMANTICS_ID,
)
from dac_her.trend_domain import (
    TREND_EVIDENCE_CONTRACT_SEMANTICS_ID,
)
from dac_her.trend_precision import PaperLocalTrendResult


HYPOTHESIS_TREND_GROUNDING_CONTRACT_SEMANTICS_ID = (
    "hypothesis_trend_grounding_contract_v1_alpha4c5a"
)

CrossStatus = Literal[
    "not_assessed",
    "repeated",
    "context_specific",
    "reversed",
    "insufficient",
]
SupportRole = Literal[
    "paper_local_only",
    "replicated_support",
    "context_dependency_signal",
    "reversal_boundary",
    "local_support_with_replication_gap",
]
PremiseScope = Literal[
    "paper_local",
    "cross_paper_replicated",
    "context_dependent",
    "reversal_boundary",
    "replication_unresolved",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GroundingSourceArtifact(StrictModel):
    role: str
    path: str
    sha256: str


class HypothesisTrendGroundingPolicy(StrictModel):
    local_trend_can_be_scoped_empirical_premise: Literal[True] = True
    repeated_can_be_cross_paper_premise: Literal[True] = True
    context_specific_can_be_context_dependency_signal: Literal[True] = True
    reversed_can_be_boundary_or_counterevidence_signal: Literal[True] = True
    insufficient_can_be_positive_cross_paper_premise: Literal[False] = False
    insufficient_can_be_gap_signal: Literal[True] = True
    correlation_can_be_causal_premise: Literal[False] = False
    trend_grounding_can_assert_causation: Literal[False] = False
    trend_grounding_can_assert_universal_relation: Literal[False] = False
    unknown_context_can_be_filled: Literal[False] = False
    majority_direction_vote_allowed: Literal[False] = False
    mechanism_can_be_promoted_to_empirical_trend: Literal[False] = False
    raw_alignment_can_be_trend_premise: Literal[False] = False
    zero_trend_yield_is_failure: Literal[False] = False


class HypothesisTrendRelationGrounding(StrictModel):
    schema_version: Literal["hypothesis-trend-relation-grounding-v1"] = (
        "hypothesis-trend-relation-grounding-v1"
    )
    grounding_id: str
    contract_semantics_id: str
    grounding_semantics_id: str
    domain_profile_id: str

    relation_id: str
    independent_variable_key: str
    dependent_observable_key: str
    control_family: str
    observable_semantics: str

    local_result_ids: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    member_trend_ids: list[str] = Field(default_factory=list)
    directions: list[str] = Field(default_factory=list)
    shapes: list[str] = Field(default_factory=list)
    evidence_kinds: list[str] = Field(default_factory=list)
    evidence_bases: list[str] = Field(default_factory=list)

    source_claim_ids: list[str] = Field(default_factory=list)
    source_measurement_ids: list[str] = Field(default_factory=list)
    source_measurement_result_ids: list[str] = Field(default_factory=list)
    source_calculation_ids: list[str] = Field(default_factory=list)
    source_node_ids: list[str] = Field(default_factory=list)

    association_only_result_ids: list[str] = Field(default_factory=list)
    source_asserted_causal_trend_ids: list[str] = Field(default_factory=list)
    source_requires_verification_trend_ids: list[str] = Field(default_factory=list)

    cross_context_assessment_id: str | None = None
    cross_context_status: CrossStatus
    pairwise_contrast_ids: list[str] = Field(default_factory=list)
    repeated_pair_ids: list[str] = Field(default_factory=list)
    reversal_pair_ids: list[str] = Field(default_factory=list)
    context_specific_pair_ids: list[str] = Field(default_factory=list)
    unresolved_pair_ids: list[str] = Field(default_factory=list)
    differentiating_dimensions: list[str] = Field(default_factory=list)
    unresolved_dimensions: list[str] = Field(default_factory=list)
    cross_context_reason_codes: list[str] = Field(default_factory=list)

    support_role: SupportRole
    premise_scope: PremiseScope

    local_empirical_premise_allowed: bool
    cross_context_replicated_premise_allowed: bool
    context_dependency_premise_allowed: bool
    reversal_counterevidence_required: bool
    replication_gap_signal_allowed: bool
    directional_cross_paper_premise_allowed: bool

    requires_context_qualification: bool
    requires_verification: bool

    causal_claim_allowed: Literal[False] = False
    universal_claim_allowed: Literal[False] = False
    majority_vote_used: Literal[False] = False
    context_filled_from_unknown: Literal[False] = False
    mechanism_promoted_to_empirical_trend: Literal[False] = False
    alignment_used_as_trend_premise: Literal[False] = False

    @model_validator(mode="after")
    def _semantic_consistency(self) -> "HypothesisTrendRelationGrounding":
        if self.contract_semantics_id != (
            HYPOTHESIS_TREND_GROUNDING_CONTRACT_SEMANTICS_ID
        ):
            raise ValueError("Hypothesis Trend grounding contract semantics mismatch.")

        if not self.local_result_ids:
            raise ValueError("Relation grounding requires local Trend results.")
        if not self.paper_ids:
            raise ValueError("Relation grounding requires paper_ids.")
        if not self.member_trend_ids:
            raise ValueError("Relation grounding requires source TrendEvidence IDs.")

        unique_fields = (
            "local_result_ids",
            "paper_ids",
            "member_trend_ids",
            "directions",
            "shapes",
            "evidence_kinds",
            "evidence_bases",
            "source_claim_ids",
            "source_measurement_ids",
            "source_measurement_result_ids",
            "source_calculation_ids",
            "source_node_ids",
            "association_only_result_ids",
            "source_asserted_causal_trend_ids",
            "source_requires_verification_trend_ids",
            "pairwise_contrast_ids",
            "repeated_pair_ids",
            "reversal_pair_ids",
            "context_specific_pair_ids",
            "unresolved_pair_ids",
            "differentiating_dimensions",
            "unresolved_dimensions",
            "cross_context_reason_codes",
        )
        for name in unique_fields:
            values = getattr(self, name)
            if values != sorted(set(values)):
                raise ValueError(f"{name} must be sorted and unique.")

        if not set(self.association_only_result_ids).issubset(
            set(self.local_result_ids)
        ):
            raise ValueError(
                "association_only_result_ids must be local-result IDs."
            )

        pair_role_union = (
            set(self.repeated_pair_ids)
            | set(self.reversal_pair_ids)
            | set(self.context_specific_pair_ids)
            | set(self.unresolved_pair_ids)
        )
        if pair_role_union != set(self.pairwise_contrast_ids):
            raise ValueError(
                "Cross-context pair-role buckets must cover all pair IDs."
            )

        expected = capabilities_for_status(
            self.cross_context_status,
            directions=self.directions,
        )
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(
                    f"{field_name} inconsistent with cross-context status: "
                    f"{getattr(self, field_name)!r} != {expected_value!r}"
                )

        if self.cross_context_status == "not_assessed":
            if self.cross_context_assessment_id is not None:
                raise ValueError(
                    "not_assessed relation cannot carry an assessment ID."
                )
            if self.pairwise_contrast_ids:
                raise ValueError(
                    "not_assessed relation cannot carry pairwise contrasts."
                )
        else:
            if not (self.cross_context_assessment_id or "").strip():
                raise ValueError(
                    "Assessed relation requires cross_context_assessment_id."
                )

        if self.cross_context_status == "reversed":
            if not self.reversal_pair_ids:
                raise ValueError("reversed grounding requires reversal pairs.")
        elif self.reversal_pair_ids:
            raise ValueError(
                "Any reversal pair requires reversed status; no majority vote."
            )

        if self.cross_context_status == "insufficient":
            if self.cross_context_replicated_premise_allowed:
                raise ValueError(
                    "insufficient cannot become replicated positive support."
                )
            if not self.replication_gap_signal_allowed:
                raise ValueError(
                    "insufficient must remain available as a replication gap."
                )

        if self.causal_claim_allowed or self.universal_claim_allowed:
            raise ValueError(
                "Trend grounding never authorizes causal or universal claims."
            )
        return self


class HypothesisTrendGroundingBundle(StrictModel):
    schema_version: Literal["hypothesis-trend-grounding-bundle-v1"] = (
        "hypothesis-trend-grounding-bundle-v1"
    )
    bundle_id: str
    bundle_sha256: str
    contract_semantics_id: str
    grounding_semantics_id: str
    domain_profile_id: str

    source_trend_semantics_id: str
    source_precision_semantics_id: str
    source_cross_context_contract_semantics_id: str | None = None
    source_cross_context_assessment_semantics_id: str | None = None

    source_artifacts: list[GroundingSourceArtifact]
    groundings: list[HypothesisTrendRelationGrounding]

    relation_count: int
    local_result_count: int
    cross_context_status_counts: dict[str, int]
    support_role_counts: dict[str, int]
    local_empirical_premise_count: int
    cross_context_replicated_premise_count: int
    context_dependency_signal_count: int
    reversal_counterevidence_count: int
    replication_gap_signal_count: int

    zero_yield: bool = False
    policy: HypothesisTrendGroundingPolicy = Field(
        default_factory=HypothesisTrendGroundingPolicy
    )

    @model_validator(mode="after")
    def _bundle_consistency(self) -> "HypothesisTrendGroundingBundle":
        if self.contract_semantics_id != (
            HYPOTHESIS_TREND_GROUNDING_CONTRACT_SEMANTICS_ID
        ):
            raise ValueError("Bundle contract semantics mismatch.")
        if self.relation_count != len(self.groundings):
            raise ValueError("relation_count mismatch.")
        if self.local_result_count != sum(
            len(row.local_result_ids) for row in self.groundings
        ):
            raise ValueError("local_result_count mismatch.")

        observed_status = Counter(
            row.cross_context_status for row in self.groundings
        )
        observed_role = Counter(row.support_role for row in self.groundings)
        if dict(sorted(observed_status.items())) != self.cross_context_status_counts:
            raise ValueError("cross_context_status_counts mismatch.")
        if dict(sorted(observed_role.items())) != self.support_role_counts:
            raise ValueError("support_role_counts mismatch.")

        observed_counts = {
            "local_empirical_premise_count": sum(
                row.local_empirical_premise_allowed
                for row in self.groundings
            ),
            "cross_context_replicated_premise_count": sum(
                row.cross_context_replicated_premise_allowed
                for row in self.groundings
            ),
            "context_dependency_signal_count": sum(
                row.context_dependency_premise_allowed
                for row in self.groundings
            ),
            "reversal_counterevidence_count": sum(
                row.reversal_counterevidence_required
                for row in self.groundings
            ),
            "replication_gap_signal_count": sum(
                row.replication_gap_signal_allowed
                for row in self.groundings
            ),
        }
        for name, observed in observed_counts.items():
            if getattr(self, name) != observed:
                raise ValueError(f"{name} mismatch.")

        if self.zero_yield != (self.local_result_count == 0):
            raise ValueError("zero_yield must exactly track local_result_count.")
        if self.zero_yield and self.groundings:
            raise ValueError("zero-yield bundle must not invent groundings.")
        return self


def _sorted_unique(values: Any) -> list[str]:
    return sorted({
        str(value)
        for value in values
        if str(value).strip()
    })


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def capabilities_for_status(
    status: str,
    *,
    directions: list[str],
) -> dict[str, Any]:
    resolved_directions = sorted({
        value
        for value in directions
        if value and value != "unspecified"
    })
    one_direction = len(resolved_directions) == 1

    if status == "repeated":
        return {
            "support_role": "replicated_support",
            "premise_scope": "cross_paper_replicated",
            "local_empirical_premise_allowed": True,
            "cross_context_replicated_premise_allowed": True,
            "context_dependency_premise_allowed": False,
            "reversal_counterevidence_required": False,
            "replication_gap_signal_allowed": False,
            "directional_cross_paper_premise_allowed": one_direction,
            "requires_context_qualification": True,
            "requires_verification": False,
        }
    if status == "context_specific":
        return {
            "support_role": "context_dependency_signal",
            "premise_scope": "context_dependent",
            "local_empirical_premise_allowed": True,
            "cross_context_replicated_premise_allowed": False,
            "context_dependency_premise_allowed": True,
            "reversal_counterevidence_required": False,
            "replication_gap_signal_allowed": False,
            "directional_cross_paper_premise_allowed": False,
            "requires_context_qualification": True,
            "requires_verification": False,
        }
    if status == "reversed":
        return {
            "support_role": "reversal_boundary",
            "premise_scope": "reversal_boundary",
            "local_empirical_premise_allowed": True,
            "cross_context_replicated_premise_allowed": False,
            "context_dependency_premise_allowed": True,
            "reversal_counterevidence_required": True,
            "replication_gap_signal_allowed": False,
            "directional_cross_paper_premise_allowed": False,
            "requires_context_qualification": True,
            "requires_verification": False,
        }
    if status == "insufficient":
        return {
            "support_role": "local_support_with_replication_gap",
            "premise_scope": "replication_unresolved",
            "local_empirical_premise_allowed": True,
            "cross_context_replicated_premise_allowed": False,
            "context_dependency_premise_allowed": False,
            "reversal_counterevidence_required": False,
            "replication_gap_signal_allowed": True,
            "directional_cross_paper_premise_allowed": False,
            "requires_context_qualification": True,
            "requires_verification": True,
        }
    if status == "not_assessed":
        return {
            "support_role": "paper_local_only",
            "premise_scope": "paper_local",
            "local_empirical_premise_allowed": True,
            "cross_context_replicated_premise_allowed": False,
            "context_dependency_premise_allowed": False,
            "reversal_counterevidence_required": False,
            "replication_gap_signal_allowed": False,
            "directional_cross_paper_premise_allowed": False,
            "requires_context_qualification": True,
            "requires_verification": False,
        }
    raise ValueError(f"Unknown cross-context status: {status!r}")


def _local_result_from_row(row: Mapping[str, Any]) -> PaperLocalTrendResult:
    values = dict(row)
    for key in (
        "member_trend_ids",
        "evidence_kinds",
        "trend_subject_ids",
        "reference_subject_ids",
        "source_claim_ids",
        "source_measurement_ids",
        "source_measurement_result_ids",
        "source_calculation_ids",
        "source_node_ids",
    ):
        values[key] = tuple(values.get(key) or ())
    return PaperLocalTrendResult(**values)


def _profile_from_row(row: Mapping[str, Any]) -> TrendContextProfile:
    values = dict(row)
    dimensions = []
    for item in values.get("dimensions") or ():
        item = dict(item)
        for key in (
            "source_values",
            "source_node_ids",
            "provenance_scopes",
        ):
            item[key] = tuple(item.get(key) or ())
        dimensions.append(TrendContextDimension(**item))
    values["dimensions"] = tuple(dimensions)
    for key in (
        "evidence_kinds",
        "member_trend_ids",
        "source_comparison_context_ids",
        "source_method_context_ids",
        "source_claim_ids",
        "source_measurement_ids",
        "source_measurement_result_ids",
        "source_calculation_ids",
        "source_node_ids",
    ):
        values[key] = tuple(values.get(key) or ())
    return TrendContextProfile(**values)


def _contrast_from_row(row: Mapping[str, Any]) -> PairwiseTrendContrast:
    values = dict(row)
    for key in (
        "matched_dimensions",
        "mismatched_dimensions",
        "unknown_dimensions",
        "ambiguous_dimensions",
        "varied_control_dimensions",
        "not_applicable_dimensions",
        "differentiating_dimensions",
        "reason_codes",
    ):
        values[key] = tuple(values.get(key) or ())
    return PairwiseTrendContrast(**values)


def _assessment_from_row(
    row: Mapping[str, Any],
) -> CrossContextTrendAssessment:
    values = dict(row)
    for key in (
        "member_result_ids",
        "paper_ids",
        "pairwise_contrast_ids",
        "positive_result_ids",
        "negative_result_ids",
        "non_monotonic_result_ids",
        "unchanged_result_ids",
        "unspecified_result_ids",
        "experimental_numeric_result_ids",
        "calculated_numeric_result_ids",
        "reported_claim_result_ids",
        "repeated_pair_ids",
        "reversal_pair_ids",
        "context_specific_pair_ids",
        "unresolved_pair_ids",
        "differentiating_dimensions",
        "unresolved_dimensions",
        "reason_codes",
    ):
        values[key] = tuple(values.get(key) or ())
    return CrossContextTrendAssessment(**values)


def build_hypothesis_trend_grounding_bundle(
    *,
    domain_profile_id: str,
    grounding_semantics_id: str,
    trend_summary: Mapping[str, Any],
    precision_summary: Mapping[str, Any],
    evidence_rows: list[Mapping[str, Any]],
    local_result_rows: list[Mapping[str, Any]],
    context_summary: Mapping[str, Any] | None,
    context_profile_rows: list[Mapping[str, Any]],
    assessment_summary: Mapping[str, Any] | None,
    assessment_rows: list[Mapping[str, Any]],
    contrast_rows: list[Mapping[str, Any]],
    source_artifacts: list[GroundingSourceArtifact],
) -> HypothesisTrendGroundingBundle:
    if trend_summary.get("structural_gate") is not True:
        raise ValueError("Trend source structural gate must pass.")
    if precision_summary.get("structural_gate") is not True:
        raise ValueError("TrendPrecision structural gate must pass.")

    trend_semantics = str(
        trend_summary.get("trend_semantics_id")
        or trend_summary.get("trend_semantics")
        or ""
    )
    precision_semantics = str(
        precision_summary.get("precision_semantics_id")
        or precision_summary.get("precision_semantics")
        or ""
    )
    if not trend_semantics or not precision_semantics:
        raise ValueError(
            "Trend/Precision summary must expose frozen semantic IDs."
        )

    local_results = [
        _local_result_from_row(row) for row in local_result_rows
    ]
    if int(
        precision_summary.get(
            "local_result_count",
            len(local_results),
        )
    ) != len(local_results):
        raise ValueError("Precision local_result_count mismatch.")

    evidence_by_id = {
        str(row.get("trend_id") or ""): dict(row)
        for row in evidence_rows
    }
    if "" in evidence_by_id:
        raise ValueError("TrendEvidence row missing trend_id.")

    for result in local_results:
        missing = set(result.member_trend_ids) - set(evidence_by_id)
        if missing:
            raise ValueError(
                f"{result.result_id} references missing TrendEvidence: "
                f"{sorted(missing)!r}"
            )

    if not local_results:
        if context_profile_rows or assessment_rows or contrast_rows:
            raise ValueError(
                "Zero local-result source cannot carry downstream "
                "CrossContext artifacts."
            )
        payload = {
            "schema_version": "hypothesis-trend-grounding-bundle-v1",
            "contract_semantics_id":
                HYPOTHESIS_TREND_GROUNDING_CONTRACT_SEMANTICS_ID,
            "grounding_semantics_id": grounding_semantics_id,
            "domain_profile_id": domain_profile_id,
            "source_trend_semantics_id": trend_semantics,
            "source_precision_semantics_id": precision_semantics,
            "source_cross_context_contract_semantics_id": None,
            "source_cross_context_assessment_semantics_id": None,
            "source_artifacts": [
                row.model_dump(mode="json") for row in source_artifacts
            ],
            "groundings": [],
            "relation_count": 0,
            "local_result_count": 0,
            "cross_context_status_counts": {},
            "support_role_counts": {},
            "local_empirical_premise_count": 0,
            "cross_context_replicated_premise_count": 0,
            "context_dependency_signal_count": 0,
            "reversal_counterevidence_count": 0,
            "replication_gap_signal_count": 0,
            "zero_yield": True,
            "policy": HypothesisTrendGroundingPolicy().model_dump(
                mode="json"
            ),
        }
        bundle_id = _stable_id(
            "hypothesis_trend_grounding",
            domain_profile_id,
            grounding_semantics_id,
            *[row.sha256 for row in source_artifacts],
        )
        payload["bundle_id"] = bundle_id
        payload["bundle_sha256"] = _sha256_json({
            k: v for k, v in payload.items()
            if k != "bundle_sha256"
        })
        return HypothesisTrendGroundingBundle(**payload)

    profiles = [_profile_from_row(row) for row in context_profile_rows]
    assessments = [
        _assessment_from_row(row) for row in assessment_rows
    ]
    contrasts = [_contrast_from_row(row) for row in contrast_rows]

    if context_summary is None or assessment_summary is None:
        raise ValueError(
            "Nonzero local Trend results require frozen CrossContext "
            "profile and assessment artifacts for alpha4c.5a."
        )
    if context_summary.get("structural_gate") is not True:
        raise ValueError("CrossContext structural gate must pass.")
    if assessment_summary.get("structural_gate") is not True:
        raise ValueError("CrossContextAssessment structural gate must pass.")

    context_contract = str(
        context_summary.get("contract_semantics_id")
        or context_summary.get("contract_semantics")
        or ""
    )
    assessment_semantics = str(
        assessment_summary.get("assessment_semantics_id")
        or assessment_summary.get("assessment_semantics")
        or ""
    )
    if context_contract != CROSS_CONTEXT_TREND_CONTRACT_SEMANTICS_ID:
        raise ValueError("CrossContext contract semantics mismatch.")
    if (
        assessment_semantics
        != CROSS_CONTEXT_TREND_ASSESSMENT_SEMANTICS_ID
    ):
        raise ValueError("CrossContext assessment semantics mismatch.")

    result_by_id = {row.result_id: row for row in local_results}
    if len(result_by_id) != len(local_results):
        raise ValueError("Duplicate PaperLocalTrendResult ID.")

    profile_by_result: dict[str, TrendContextProfile] = {}
    for profile in profiles:
        if profile.local_result_id not in result_by_id:
            raise ValueError(
                "TrendContextProfile references unknown local result: "
                f"{profile.local_result_id}"
            )
        if profile.local_result_id in profile_by_result:
            raise ValueError(
                "Multiple context profiles reference one local result."
            )
        profile_by_result[profile.local_result_id] = profile
    if set(profile_by_result) != set(result_by_id):
        raise ValueError(
            "CrossContext profile coverage must exactly match local results."
        )

    contrast_by_id = {row.contrast_id: row for row in contrasts}
    if len(contrast_by_id) != len(contrasts):
        raise ValueError("Duplicate PairwiseTrendContrast ID.")

    assessment_by_relation: dict[str, CrossContextTrendAssessment] = {}
    for assessment in assessments:
        if assessment.relation_id in assessment_by_relation:
            raise ValueError(
                "Multiple assessments found for one TrendRelation."
            )
        assessment_by_relation[assessment.relation_id] = assessment
        missing_results = set(assessment.member_result_ids) - set(result_by_id)
        if missing_results:
            raise ValueError(
                "Assessment references unknown local results: "
                f"{sorted(missing_results)!r}"
            )
        missing_pairs = (
            set(assessment.pairwise_contrast_ids)
            - set(contrast_by_id)
        )
        if missing_pairs:
            raise ValueError(
                "Assessment references unknown contrasts: "
                f"{sorted(missing_pairs)!r}"
            )

    grouped: dict[str, list[PaperLocalTrendResult]] = {}
    for result in local_results:
        profile = profile_by_result[result.result_id]
        grouped.setdefault(profile.relation_id, []).append(result)

    if set(grouped) != set(assessment_by_relation):
        raise ValueError(
            "Every frozen TrendRelation must have exactly one "
            "CrossContextTrendAssessment."
        )

    groundings: list[HypothesisTrendRelationGrounding] = []

    for relation_id in sorted(grouped):
        results = sorted(
            grouped[relation_id],
            key=lambda row: (row.paper_id, row.result_id),
        )
        assessment = assessment_by_relation[relation_id]
        profiles_for_relation = [
            profile_by_result[row.result_id] for row in results
        ]
        first = profiles_for_relation[0]

        for profile in profiles_for_relation[1:]:
            for field in (
                "independent_variable_key",
                "dependent_observable_key",
                "control_family",
                "observable_semantics",
            ):
                if getattr(profile, field) != getattr(first, field):
                    raise ValueError(
                        f"Relation {relation_id} mixes {field}."
                    )

        member_trend_ids = _sorted_unique([
            trend_id
            for result in results
            for trend_id in result.member_trend_ids
        ])

        evidence_rows_for_relation = [
            evidence_by_id[trend_id]
            for trend_id in member_trend_ids
        ]
        bases = _sorted_unique([
            row.get("evidence_basis", "")
            for row in evidence_rows_for_relation
        ])
        association_result_ids = _sorted_unique([
            result.result_id
            for result in results
            if any(
                evidence_by_id[trend_id].get("evidence_basis")
                == "reported_correlation"
                for trend_id in result.member_trend_ids
            )
        ])
        source_asserted = _sorted_unique([
            row.get("trend_id", "")
            for row in evidence_rows_for_relation
            if row.get("causal_status") == "source_asserted"
        ])
        verification_trends = _sorted_unique([
            row.get("trend_id", "")
            for row in evidence_rows_for_relation
            if bool(row.get("requires_verification", False))
        ])

        capabilities = capabilities_for_status(
            assessment.status,
            directions=_sorted_unique([
                result.direction for result in results
            ]),
        )
        if verification_trends:
            capabilities["requires_verification"] = True

        grounding_id = _stable_id(
            "hypothesis_trend_relation",
            grounding_semantics_id,
            relation_id,
            *[row.result_id for row in results],
            assessment.assessment_id,
        )

        groundings.append(
            HypothesisTrendRelationGrounding(
                grounding_id=grounding_id,
                contract_semantics_id=
                    HYPOTHESIS_TREND_GROUNDING_CONTRACT_SEMANTICS_ID,
                grounding_semantics_id=grounding_semantics_id,
                domain_profile_id=domain_profile_id,
                relation_id=relation_id,
                independent_variable_key=
                    first.independent_variable_key,
                dependent_observable_key=
                    first.dependent_observable_key,
                control_family=first.control_family,
                observable_semantics=first.observable_semantics,
                local_result_ids=_sorted_unique([
                    row.result_id for row in results
                ]),
                paper_ids=_sorted_unique([
                    row.paper_id for row in results
                ]),
                member_trend_ids=member_trend_ids,
                directions=_sorted_unique([
                    row.direction for row in results
                ]),
                shapes=_sorted_unique([
                    row.shape for row in results
                ]),
                evidence_kinds=_sorted_unique([
                    kind
                    for row in results
                    for kind in row.evidence_kinds
                ]),
                evidence_bases=bases,
                source_claim_ids=_sorted_unique([
                    value
                    for row in results
                    for value in row.source_claim_ids
                ]),
                source_measurement_ids=_sorted_unique([
                    value
                    for row in results
                    for value in row.source_measurement_ids
                ]),
                source_measurement_result_ids=_sorted_unique([
                    value
                    for row in results
                    for value in row.source_measurement_result_ids
                ]),
                source_calculation_ids=_sorted_unique([
                    value
                    for row in results
                    for value in row.source_calculation_ids
                ]),
                source_node_ids=_sorted_unique([
                    value
                    for row in results
                    for value in row.source_node_ids
                ]),
                association_only_result_ids=association_result_ids,
                source_asserted_causal_trend_ids=source_asserted,
                source_requires_verification_trend_ids=
                    verification_trends,
                cross_context_assessment_id=
                    assessment.assessment_id,
                cross_context_status=assessment.status,
                pairwise_contrast_ids=_sorted_unique(
                    assessment.pairwise_contrast_ids
                ),
                repeated_pair_ids=_sorted_unique(
                    assessment.repeated_pair_ids
                ),
                reversal_pair_ids=_sorted_unique(
                    assessment.reversal_pair_ids
                ),
                context_specific_pair_ids=_sorted_unique(
                    assessment.context_specific_pair_ids
                ),
                unresolved_pair_ids=_sorted_unique(
                    assessment.unresolved_pair_ids
                ),
                differentiating_dimensions=_sorted_unique(
                    assessment.differentiating_dimensions
                ),
                unresolved_dimensions=_sorted_unique(
                    assessment.unresolved_dimensions
                ),
                cross_context_reason_codes=_sorted_unique(
                    assessment.reason_codes
                ),
                **capabilities,
                causal_claim_allowed=False,
                universal_claim_allowed=False,
                majority_vote_used=False,
                context_filled_from_unknown=False,
                mechanism_promoted_to_empirical_trend=False,
                alignment_used_as_trend_premise=False,
            )
        )

    status_counts = dict(sorted(Counter(
        row.cross_context_status for row in groundings
    ).items()))
    role_counts = dict(sorted(Counter(
        row.support_role for row in groundings
    ).items()))

    payload = {
        "schema_version": "hypothesis-trend-grounding-bundle-v1",
        "contract_semantics_id":
            HYPOTHESIS_TREND_GROUNDING_CONTRACT_SEMANTICS_ID,
        "grounding_semantics_id": grounding_semantics_id,
        "domain_profile_id": domain_profile_id,
        "source_trend_semantics_id": trend_semantics,
        "source_precision_semantics_id": precision_semantics,
        "source_cross_context_contract_semantics_id":
            context_contract,
        "source_cross_context_assessment_semantics_id":
            assessment_semantics,
        "source_artifacts": [
            row.model_dump(mode="json") for row in source_artifacts
        ],
        "groundings": [
            row.model_dump(mode="json") for row in groundings
        ],
        "relation_count": len(groundings),
        "local_result_count": len(local_results),
        "cross_context_status_counts": status_counts,
        "support_role_counts": role_counts,
        "local_empirical_premise_count": sum(
            row.local_empirical_premise_allowed for row in groundings
        ),
        "cross_context_replicated_premise_count": sum(
            row.cross_context_replicated_premise_allowed
            for row in groundings
        ),
        "context_dependency_signal_count": sum(
            row.context_dependency_premise_allowed
            for row in groundings
        ),
        "reversal_counterevidence_count": sum(
            row.reversal_counterevidence_required
            for row in groundings
        ),
        "replication_gap_signal_count": sum(
            row.replication_gap_signal_allowed
            for row in groundings
        ),
        "zero_yield": False,
        "policy": HypothesisTrendGroundingPolicy().model_dump(
            mode="json"
        ),
    }
    bundle_id = _stable_id(
        "hypothesis_trend_grounding",
        domain_profile_id,
        grounding_semantics_id,
        *[row.sha256 for row in source_artifacts],
    )
    payload["bundle_id"] = bundle_id
    payload["bundle_sha256"] = _sha256_json({
        k: v for k, v in payload.items()
        if k != "bundle_sha256"
    })
    return HypothesisTrendGroundingBundle(**payload)
